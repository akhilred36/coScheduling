"""
CoScheduling Runs Parser Module (Parallel Version)

This module provides utilities to parse coscheduling experiment run directories
and extract performance metrics from mpiP output files using parallel processing.
"""

import re
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
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


class CoSchedulingRunsParserParallel:
    """
    Parser for coscheduling experiment run directories (parallel version).
    
    This class parses directories containing mpiP output files and extracts
    performance metrics for each run configuration using parallel processing.
    """
    
    def __init__(self, base_path: Optional[Union[str, Path]] = None):
        """
        Initialize the CoScheduling Runs Parser.
        
        Args:
            base_path: Optional path to the base directory containing run subdirectories.
        """
        self.base_path: Optional[Path] = Path(base_path) if base_path else None
        self._runs: List[RunMetrics] = []
    
    def parse_isolated_parallel(self, directory: Union[str, Path], max_workers: int = 10) -> pd.DataFrame:
        """
        Parse an isolated experiment directory and extract metrics in parallel.
        
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
            
        Raises:
            ValueError: If any subdirectory fails to parse correctly.
        """
        directory = Path(directory)
        self.base_path = directory
        self._runs = []
        
        # Collect all subdirectories to parse
        subdirs_to_parse = []
        total_subdirs = 0
        
        for subdir in directory.iterdir():
            if not subdir.is_dir():
                continue
            
            total_subdirs += 1
            
            match = re.match(r'^(\d+)_(.+)_(\d+)$', subdir.name)
            if not match:
                continue
            
            num_nodes = int(match.group(1))
            app = match.group(2)
            run_iteration = int(match.group(3))
            
            mpip_profiles_dir = subdir / "mpip_profiles"
            if not mpip_profiles_dir.exists():
                continue
            
            mpiP_files = list(mpip_profiles_dir.glob("*.mpiP"))
            if not mpiP_files:
                continue
            
            subdirs_to_parse.append({
                'subdir': subdir,
                'num_nodes': num_nodes,
                'app': app,
                'run_iteration': run_iteration,
                'mpiP_files': mpiP_files
            })
        
        # Parse each subdirectory in parallel
        parsed_results = []
        parse_errors = []
        
        def parse_subdir(subdir_info):
            try:
                subdir = subdir_info['subdir']
                
                mpiP_file = subdir_info['mpiP_files'][0]
                parser = MPIPParser(mpiP_file)
                
                if parser.mpi_time_df is not None:
                    app_time_avg = parser.mpi_time_df['app_time'].mean()
                    mpi_time_avg = parser.mpi_time_df['mpi_time'].mean()
                else:
                    raise Exception(f"{subdir.name}: mpi_time_df is None")
                
                total_messages_sent = 0
                if parser.aggregate_sent_df is not None:
                    total_messages_sent = parser.aggregate_sent_df['count'].sum()
                
                return RunMetrics(
                    app=subdir_info['app'],
                    num_nodes=subdir_info['num_nodes'],
                    run_iteration=subdir_info['run_iteration'],
                    app_time_average=app_time_avg,
                    mpi_time_average=mpi_time_avg,
                    total_messages_sent=total_messages_sent
                )
                
            except Exception as e:
                return e
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(parse_subdir, subdirs_to_parse))
        
        for result in results:
            if isinstance(result, Exception):
                parse_errors.append(str(result))
            else:
                parsed_results.append(result)
        
        # Check for failures
        failed_subdirs = parse_errors
        parsed_subdirs = len(parsed_results)
        
        if failed_subdirs:
            error_msg = f"Failed to parse {len(failed_subdirs)} out of {total_subdirs} subdirectories:\n"
            for failure in failed_subdirs:
                error_msg += f"  - {failure}\n"
            raise ValueError(error_msg)
        
        # Convert to DataFrame
        df = pd.DataFrame({
            "App": [r.app for r in parsed_results],
            "Num Nodes": [r.num_nodes for r in parsed_results],
            "Run Iteration": [r.run_iteration for r in parsed_results],
            "App Time Average": [r.app_time_average for r in parsed_results],
            "MPI Time Average": [r.mpi_time_average for r in parsed_results],
            "Total Messages Sent": [r.total_messages_sent for r in parsed_results]
        })
        
        return df
    
    def parse_coscheduled_parallel(self, directory: Union[str, Path], max_workers: int = 10) -> pd.DataFrame:
        """
        Parse a coscheduled experiment directory and extract metrics for paired applications in parallel.
        
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
            
        Raises:
            ValueError: If any subdirectory fails to parse correctly.
        """
        directory = Path(directory)
        self.base_path = directory
        self._runs = []
        
        # Load the run config to map exec names to app names
        run_config_path = Path("../run_configs/8_nodes.json")
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
        
        # Collect all subdirectories to parse
        subdirs_to_parse = []
        total_subdirs = 0
        
        for subdir in directory.iterdir():
            if not subdir.is_dir():
                continue
            
            total_subdirs += 1
            
            match = re.match(r'^(\d+)_(.+)_(.+)_(\d+)$', subdir.name)
            if not match:
                continue
            
            num_nodes = int(match.group(1))
            app_a_name = match.group(2)
            app_b_name = match.group(3)
            run_iteration = int(match.group(4))
            
            mpip_profiles_dir = subdir / "mpip_profiles"
            if not mpip_profiles_dir.exists():
                continue
            
            mpiP_files = list(mpip_profiles_dir.glob("*.mpiP"))
            if len(mpiP_files) != 2:
                continue
            
            subdirs_to_parse.append({
                'subdir': subdir,
                'num_nodes': num_nodes,
                'app_a_name': app_a_name,
                'app_b_name': app_b_name,
                'run_iteration': run_iteration,
                'mpiP_files': mpiP_files,
                'exec_to_app': exec_to_app
            })
        
       # Parse each subdirectory in parallel
        parsed_results = []
        parse_errors = []
        
        def parse_subdir(subdir_info):
            try:
                subdir = subdir_info['subdir']
                
                mpiP_files = subdir_info['mpiP_files']
                exec_to_app = subdir_info['exec_to_app']
                
                app_a_metrics = None
                app_b_metrics = None
                
                for mpiP_file in mpiP_files:
                    filename = mpiP_file.stem
                    parts = filename.split('.')
                    
                    exec_name = None
                    if filename in exec_to_app:
                        exec_name = filename
                    else:
                        for i in range(len(parts), 0, -1):
                            candidate = '.'.join(parts[:i])
                            if candidate in exec_to_app:
                                exec_name = candidate
                                break
                    
                    if not exec_name:
                        raise Exception(f"{subdir.name}: Could not find exec_name for '{filename}'")
                    
                    app_name = exec_to_app[exec_name]
                    parser = MPIPParser(mpiP_file)
                    
                    if parser.mpi_time_df is not None:
                        app_time_avg = parser.mpi_time_df['app_time'].mean()
                        mpi_time_avg = parser.mpi_time_df['mpi_time'].mean()
                    else:
                        raise Exception(f"{subdir.name}: mpi_time_df is None for '{filename}'")
                    
                    total_messages_sent = 0
                    if parser.aggregate_sent_df is not None:
                        total_messages_sent = int(parser.aggregate_sent_df['count'].sum())
                    
                    metrics = RunMetrics(
                        app=app_name,
                        num_nodes=subdir_info['num_nodes'],
                        run_iteration=subdir_info['run_iteration'],
                        app_time_average=app_time_avg,
                        mpi_time_average=mpi_time_avg,
                        total_messages_sent=total_messages_sent
                    )
                    
                    if app_name == subdir_info['app_a_name']:
                        if app_a_metrics is None:
                            app_a_metrics = metrics
                        elif app_b_metrics is None:
                            app_b_metrics = metrics
                    elif app_name == subdir_info['app_b_name']:
                        app_b_metrics = metrics
                
                if app_a_metrics is not None and app_b_metrics is not None:
                    return {
                        "Num Nodes": subdir_info['num_nodes'],
                        "App A": subdir_info['app_a_name'],
                        "App B": subdir_info['app_b_name'],
                        "Run Iteration": subdir_info['run_iteration'],
                        "App A Time Average": app_a_metrics.app_time_average,
                        "App B Time Average": app_b_metrics.app_time_average,
                        "App A MPI Time Average": app_a_metrics.mpi_time_average,
                        "App B MPI Time Average": app_b_metrics.mpi_time_average,
                        "App A Total Messages Sent": app_a_metrics.total_messages_sent,
                        "App B Total Messages Sent": app_b_metrics.total_messages_sent
                    }
                else:
                    missing_apps = []
                    if app_a_metrics is None:
                        missing_apps.append(subdir_info['app_a_name'])
                    if app_b_metrics is None:
                        missing_apps.append(subdir_info['app_b_name'])
                    raise Exception(f"{subdir.name}: could not find metrics for app(s) {missing_apps}")
            
            except Exception as e:
                return e
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(parse_subdir, subdirs_to_parse))
        
        for result in results:
            if isinstance(result, Exception):
                parse_errors.append(str(result))
            else:
                parsed_results.append(result)
        
        # Check for failures
        failed_subdirs = parse_errors
        parsed_subdirs = len(parsed_results)
        
        if failed_subdirs:
            error_msg = f"Failed to parse {len(failed_subdirs)} out of {total_subdirs} subdirectories:\n"
            for failure in failed_subdirs:
                error_msg += f"  - {failure}\n"
            raise ValueError(error_msg)
        
        # Convert to DataFrame
        df = pd.DataFrame(parsed_results)
        
        return df
    
    def parse_inhibitor_coscheduled_parallel(self, directory: Union[str, Path], max_workers: int = 10) -> pd.DataFrame:
        """
        Parse an inhibitor coscheduled experiment directory and extract metrics in parallel.
        
        The directory structure is expected to be:
        <directory>/
            <numNodes>_<app>_inhib_<Inhib Message Size>_<Inhib Wait Time (us)>_-1_<Inhib Comm Mode>_<Inhib Comm Sparsity>_<runIteration>/
                mpip_profiles/
                    <filename>.mpiP
            ...
        
        Args:
            directory: Path to the experiment data directory.
            
        Returns:
            pandas DataFrame with columns:
            - "App": Application name
            - "Num Nodes": Number of nodes used
            - "Inhib Message Size": Inhibitor message size
            - "Inhib Wait Time (us)": Inhibitor wait time in microseconds
            - "Inhib Comm Sparsity": Inhibitor communication sparsity
            - "Inhib Comm Mode": Inhibitor communication mode
            - "Run Iteration": Run iteration number
            - "App Time Average": Average AppTime across all tasks (seconds)
            - "MPI Time Average": Average MPITime across all tasks (seconds)
            - "Total Messages Sent": Sum of all messages sent
            
        Raises:
            ValueError: If any subdirectory fails to parse correctly.
        """
        directory = Path(directory)
        self.base_path = directory
        self._runs = []
        
        # Collect all subdirectories to parse
        subdirs_to_parse = []
        total_subdirs = 0
        
        for subdir in directory.iterdir():
            if not subdir.is_dir():
                continue
            
            total_subdirs += 1
            
            match = re.match(
                r'^(\d+)_(.+)_inhib_(\d+)_(\d+)_-1_(\w+)_(\d+\.?\d*)_(\d+)$',
                subdir.name
            )
            if not match:
                continue
            
            num_nodes = int(match.group(1))
            app = match.group(2)
            inhib_message_size = int(match.group(3))
            inhib_wait_time_us = int(match.group(4))
            inhib_comm_mode = match.group(5)
            inhib_comm_sparsity = float(match.group(6))
            run_iteration = int(match.group(7))
            
            mpip_profiles_dir = subdir / "mpip_profiles"
            if not mpip_profiles_dir.exists():
                continue
            
            mpiP_files = list(mpip_profiles_dir.glob("*.mpiP"))
            if not mpiP_files:
                continue
            
            subdirs_to_parse.append({
                'subdir': subdir,
                'num_nodes': num_nodes,
                'app': app,
                'inhib_message_size': inhib_message_size,
                'inhib_wait_time_us': inhib_wait_time_us,
                'inhib_comm_mode': inhib_comm_mode,
                'inhib_comm_sparsity': inhib_comm_sparsity,
                'run_iteration': run_iteration,
                'mpiP_files': mpiP_files
            })
        
      # Parse each subdirectory in parallel
        parsed_results = []
        parse_errors = []
        
        def parse_subdir(subdir_info):
            try:
                subdir = subdir_info['subdir']
                
                mpiP_file = subdir_info['mpiP_files'][0]
                parser = MPIPParser(mpiP_file)
                
                if parser.mpi_time_df is not None:
                    app_time_avg = parser.mpi_time_df['app_time'].mean()
                    mpi_time_avg = parser.mpi_time_df['mpi_time'].mean()
                else:
                    raise Exception(f"{subdir.name}: mpi_time_df is None")
                
                total_messages_sent = 0
                if parser.aggregate_sent_df is not None:
                    total_messages_sent = parser.aggregate_sent_df['count'].sum()
                
                return {
                    "App": subdir_info['app'],
                    "Num Nodes": subdir_info['num_nodes'],
                    "Inhib Message Size": subdir_info['inhib_message_size'],
                    "Inhib Wait Time (us)": subdir_info['inhib_wait_time_us'],
                    "Inhib Comm Sparsity": subdir_info['inhib_comm_sparsity'],
                    "Inhib Comm Mode": subdir_info['inhib_comm_mode'],
                    "Run Iteration": subdir_info['run_iteration'],
                    "App Time Average": app_time_avg,
                    "MPI Time Average": mpi_time_avg,
                    "Total Messages Sent": total_messages_sent
                }
                
            except Exception as e:
                return e
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(parse_subdir, subdirs_to_parse))
        
        for result in results:
            if isinstance(result, Exception):
                parse_errors.append(str(result))
            else:
                parsed_results.append(result)
        
        # Check for failures
        failed_subdirs = parse_errors
        parsed_subdirs = len(parsed_results)
        
        if failed_subdirs:
            error_msg = f"Failed to parse {len(failed_subdirs)} out of {total_subdirs} subdirectories:\n"
            for failure in failed_subdirs:
                error_msg += f"  - {failure}\n"
            raise ValueError(error_msg)
        
        # Convert to DataFrame
        df = pd.DataFrame(parsed_results)
        
        return df
