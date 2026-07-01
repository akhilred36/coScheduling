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
from tqdm import tqdm

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
    
    def parse_isolated(self, directory: Union[str, Path], progress=True) -> pd.DataFrame:
        """
        Parse an isolated experiment directory and extract metrics including total_bytes.
        
        The directory structure is expected to be:
        <directory>/
            <numNodes>_<app>_<runIteration>/
                mpip_profiles/
                    <filename>.mpiP
            ...
        
        Args:
            directory: Path to the experiment data directory.
            progress: Whether to show progress bars.
            
        Returns:
            pandas DataFrame with columns:
            - "App": Application name
            - "Num Nodes": Number of nodes used
            - "Run Iteration": Run iteration number
            - "App Time Average": Average AppTime across all tasks (seconds)
            - "MPI Time Average": Average MPITime across all tasks (seconds)
            - "Total Messages Sent": Sum of all messages sent
            - "Total Bytes Sent": Sum of all total_bytes from aggregate_sent_df
            
        Raises:
            ValueError: If any subdirectory fails to parse correctly.
        """
        directory = Path(directory)
        self.base_path = directory
        self._runs = []
        
        # Phase 1: Collect all file paths and metadata first
        total_subdirs = 0
        parsed_subdirs = 0
        failed_subdirs = []
        collected_files = []
        
        for subdir in tqdm(directory.iterdir(), desc="Phase 1: Collecting files", disable=not progress):
            if not subdir.is_dir():
                continue
            
            total_subdirs += 1
            
            match = re.match(r'^(\d+)_(.+)_(\d+)$', subdir.name)
            if not match:
                failed_subdirs.append(f"{subdir.name}: failed to match directory pattern")
                continue
            
            num_nodes = int(match.group(1))
            app = match.group(2)
            run_iteration = int(match.group(3))
            
            mpip_profiles_dir = subdir / "mpip_profiles"
            if not mpip_profiles_dir.exists():
                failed_subdirs.append(f"{subdir.name}: mpip_profiles directory not found")
                continue
            
            mpiP_files = list(mpip_profiles_dir.glob("*.mpiP"))
            if not mpiP_files:
                failed_subdirs.append(f"{subdir.name}: no .mpiP files found")
                continue
            
            mpiP_file = mpiP_files[0]
            collected_files.append({
                'subdir_name': subdir.name,
                'mpiP_file': mpiP_file,
                'num_nodes': num_nodes,
                'app': app,
                'run_iteration': run_iteration
            })
        
        # Phase 2: Parse all collected files
        for file_info in tqdm(collected_files, desc="Phase 2: Parsing files", disable=not progress):
            parser = MPIPParser(file_info['mpiP_file'])
            
            if parser.mpi_time_df is not None:
                app_time_avg = parser.mpi_time_df['app_time'].mean()
                mpi_time_avg = parser.mpi_time_df['mpi_time'].mean()
            else:
                failed_subdirs.append(f"{file_info['subdir_name']}: mpi_time_df is None")
                continue
            
            total_messages_sent = 0
            total_bytes = 0
            if parser.aggregate_sent_df is not None:
                total_messages_sent = parser.aggregate_sent_df['count'].sum()
                total_bytes = parser.aggregate_sent_df['total_bytes'].sum()
            
            metrics = RunMetrics(
                app=file_info['app'],
                num_nodes=file_info['num_nodes'],
                run_iteration=file_info['run_iteration'],
                app_time_average=app_time_avg,
                mpi_time_average=mpi_time_avg,
                total_messages_sent=total_messages_sent
            )
            self._runs.append(metrics)
            parsed_subdirs += 1
        
        # Check for failures and raise error if any
        if failed_subdirs:
            error_msg = f"Failed to parse {len(failed_subdirs)} out of {total_subdirs} subdirectories:\n"
            for failure in failed_subdirs:
                error_msg += f"  - {failure}\n"
            raise ValueError(error_msg)
        
        # Convert to DataFrame
        df = pd.DataFrame({
            "App": [r.app for r in self._runs],
            "Num Nodes": [r.num_nodes for r in self._runs],
            "Run Iteration": [r.run_iteration for r in self._runs],
            "App Time Average": [r.app_time_average for r in self._runs],
            "MPI Time Average": [r.mpi_time_average for r in self._runs],
            "Total Messages Sent": [r.total_messages_sent for r in self._runs]
        })
        
        total_bytes_list = []
        for file_info in collected_files:
            parser = MPIPParser(file_info['mpiP_file'])
            if parser.aggregate_sent_df is not None:
                total_bytes_list.append(parser.aggregate_sent_df['total_bytes'].sum())
            else:
                total_bytes_list.append(0)
        
        df["Total Bytes Sent"] = total_bytes_list
        
        return df
    
    def parse_coscheduled(self, directory: Union[str, Path], progress=True) -> pd.DataFrame:
        """
        Parse a coscheduled experiment directory and extract metrics for paired applications including total_bytes.
        
        The directory structure is expected to be:
        <directory>/
            <numNodes>_<appA>_<appB>_<runIteration>/
                mpip_profiles/
                    <execA>.mpiP
                    <execB>.mpiP
            ...
        
        Args:
            directory: Path to the experiment data directory.
            progress: Whether to show progress bars.
            
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
            - "App A Total Bytes Sent": Sum of total_bytes for App A
            - "App B Total Bytes Sent": Sum of total_bytes for App B
            
        Raises:
            ValueError: If any subdirectory fails to parse correctly.
        """
        directory = Path(directory)
        self.base_path = directory
        self._runs = []
        
        run_config_path = Path("../run_configs/8_nodes.json")
        if not run_config_path.exists():
            raise FileNotFoundError(f"Run config file not found: {run_config_path}")
        
        import json
        with open(run_config_path, 'r') as f:
            run_config = json.load(f)
        
        exec_to_app = {}
        for app_name, app_info in run_config.get("apps", {}).items():
            exec_name = app_info.get("exec")
            if exec_name:
                exec_to_app[exec_name] = app_name
        
        total_subdirs = 0
        parsed_subdirs = 0
        failed_subdirs = []
        collected_files = []
        
        for subdir in tqdm(directory.iterdir(), desc="Phase 1: Collecting files", disable=not progress):
            if not subdir.is_dir():
                continue
            
            total_subdirs += 1
            
            match = re.match(r'^(\d+)_(.+)_(.+)_(\d+)$', subdir.name)
            if not match:
                failed_subdirs.append(f"{subdir.name}: failed to match directory pattern")
                continue
            
            num_nodes = int(match.group(1))
            app_a_name = match.group(2)
            app_b_name = match.group(3)
            run_iteration = int(match.group(4))
            
            mpip_profiles_dir = subdir / "mpip_profiles"
            if not mpip_profiles_dir.exists():
                failed_subdirs.append(f"{subdir.name}: mpip_profiles directory not found")
                continue
            
            mpiP_files = list(mpip_profiles_dir.glob("*.mpiP"))
            if len(mpiP_files) != 2:
                failed_subdirs.append(f"{subdir.name}: expected 2 .mpiP files, found {len(mpiP_files)}")
                continue
            
            collected_files.append({
                'subdir_name': subdir.name,
                'mpiP_files': mpiP_files,
                'num_nodes': num_nodes,
                'app_a_name': app_a_name,
                'app_b_name': app_b_name,
                'run_iteration': run_iteration
            })
        
        for file_info in tqdm(collected_files, desc="Phase 2: Parsing files", disable=not progress):
            app_a_metrics = None
            app_b_metrics = None
            app_a_total_bytes = 0
            app_b_total_bytes = 0
            parse_errors = []
            
            for mpiP_file in file_info['mpiP_files']:
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
                    parse_errors.append(f"Could not find exec_name for '{filename}' in exec_to_app mapping")
                    continue
                
                app_name = exec_to_app[exec_name]
                parser = MPIPParser(mpiP_file)
                
                if parser.mpi_time_df is not None:
                    app_time_avg = parser.mpi_time_df['app_time'].mean()
                    mpi_time_avg = parser.mpi_time_df['mpi_time'].mean()
                else:
                    parse_errors.append(f"mpi_time_df is None for '{filename}'")
                    continue
                
                total_messages_sent = 0
                total_bytes = 0
                if parser.aggregate_sent_df is not None:
                    total_messages_sent = int(parser.aggregate_sent_df['count'].sum())
                    total_bytes = parser.aggregate_sent_df['total_bytes'].sum()
                
                metrics = RunMetrics(
                    app=app_name,
                    num_nodes=file_info['num_nodes'],
                    run_iteration=file_info['run_iteration'],
                    app_time_average=app_time_avg,
                    mpi_time_average=mpi_time_avg,
                    total_messages_sent=total_messages_sent
                )
                
                if app_name == file_info['app_a_name']:
                    if app_a_metrics is None:
                        app_a_metrics = metrics
                    elif app_b_metrics is None:
                        app_b_metrics = metrics
                elif app_name == file_info['app_b_name']:
                    app_b_metrics = metrics
                
                if app_name == file_info['app_a_name']:
                    app_a_total_bytes = total_bytes
                elif app_name == file_info['app_b_name']:
                    app_b_total_bytes = total_bytes
            
            if parse_errors:
                failed_subdirs.append(f"{file_info['subdir_name']}: {'; '.join(parse_errors)}")
                continue
            
            if app_a_metrics is not None and app_b_metrics is not None:
                self._runs.append({
                    "Num Nodes": file_info['num_nodes'],
                    "App A": file_info['app_a_name'],
                    "App B": file_info['app_b_name'],
                    "Run Iteration": file_info['run_iteration'],
                    "App A Time Average": app_a_metrics.app_time_average,
                    "App B Time Average": app_b_metrics.app_time_average,
                    "App A MPI Time Average": app_a_metrics.mpi_time_average,
                    "App B MPI Time Average": app_b_metrics.mpi_time_average,
                    "App A Total Messages Sent": app_a_metrics.total_messages_sent,
                    "App B Total Messages Sent": app_b_metrics.total_messages_sent,
                    "App A Total Bytes Sent": app_a_total_bytes,
                    "App B Total Bytes Sent": app_b_total_bytes
                })
                parsed_subdirs += 1
            else:
                missing_apps = []
                if app_a_metrics is None:
                    missing_apps.append(file_info['app_a_name'])
                if app_b_metrics is None:
                    missing_apps.append(file_info['app_b_name'])
                failed_subdirs.append(f"{file_info['subdir_name']}: could not find metrics for app(s) {missing_apps}")
        
        # Check for failures and raise error if any
        if failed_subdirs:
            error_msg = f"Failed to parse {len(failed_subdirs)} out of {total_subdirs} subdirectories:\n"
            for failure in failed_subdirs:
                error_msg += f"  - {failure}\n"
            raise ValueError(error_msg)
        
        df = pd.DataFrame(self._runs)
        
        return df
    
    def parse_inhibitor_coscheduled(self, directory: Union[str, Path], progress=True) -> pd.DataFrame:
        """
        Parse an inhibitor coscheduled experiment directory and extract metrics including total_bytes.
        
        The directory structure is expected to be:
        <directory>/
            <numNodes>_<app>_inhib_<Inhib Message Size>_<Inhib Wait Time (us)>_d_<Inhib Comm Sparsity>_<runIteration>/
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
            - "Run Iteration": Run iteration number
            - "App Time Average": Average AppTime across all tasks (seconds)
            - "MPI Time Average": Average MPITime across all tasks (seconds)
            - "Total Messages Sent": Sum of all messages sent
            - "Total Bytes Sent": Sum of all total_bytes from aggregate_sent_df
            
        Raises:
            ValueError: If any subdirectory fails to parse correctly.
        """
        directory = Path(directory)
        self.base_path = directory
        self._runs = []
        
        # Phase 1: Collect all file paths and metadata first
        # This allows the OS to cache all file reads before parsing begins
        total_subdirs = 0
        parsed_subdirs = 0
        failed_subdirs = []
        collected_files = []  # List of (subdir_name, mpiP_file_path, parsed_metadata)
        
        # First pass: collect all valid file paths and parse directory names
        for subdir in tqdm(directory.iterdir(), desc="Phase 1: Collecting files", disable=not progress):
            if not subdir.is_dir():
                continue
            
            total_subdirs += 1
            
            # Parse the directory name
            # Format: <numNodes>_<app>_inhib_<Inhib Message Size>_<Inhib Wait Time (us)>_d_<Inhib Comm Sparsity>_<runIteration>
            # Note: Inhib Comm Sparsity can be a float (e.g., 0.2), so we use \d+\.?\d* to match it
            match = re.match(
                r'^(\d+)_(.+)_inhib_(\d+)_(\d+)_d_(\d+\.?\d*)_(\d+)$',
                subdir.name
            )
            if not match:
                failed_subdirs.append(f"{subdir.name}: failed to match directory pattern")
                continue
            
            num_nodes = int(match.group(1))
            app = match.group(2)
            inhib_message_size = int(match.group(3))
            inhib_wait_time_us = int(match.group(4))
            inhib_comm_sparsity = float(match.group(5))
            run_iteration = int(match.group(6))
            
            # Find the .mpiP file in mpip_profiles subdirectory
            mpip_profiles_dir = subdir / "mpip_profiles"
            if not mpip_profiles_dir.exists():
                failed_subdirs.append(f"{subdir.name}: mpip_profiles directory not found")
                continue
            
            mpiP_files = list(mpip_profiles_dir.glob("*.mpiP"))
            if not mpiP_files:
                failed_subdirs.append(f"{subdir.name}: no .mpiP files found")
                continue
            
            # Store file path and metadata for later processing
                mpiP_file = mpiP_files[0]
            collected_files.append({
                'subdir_name': subdir.name,
                'mpiP_files': mpiP_files,
                'num_nodes': num_nodes,
                'app': app,
                'inhib_message_size': inhib_message_size,
                'inhib_wait_time_us': inhib_wait_time_us,
                'inhib_comm_sparsity': inhib_comm_sparsity,
                'run_iteration': run_iteration
            })
        
        # Phase 2: Parse all collected files
        for file_info in tqdm(collected_files, desc="Phase 2: Parsing files", disable=not progress):
            parser = MPIPParser(file_info['mpiP_file'])
            
            # Extract metrics from MPI Time section
            if parser.mpi_time_df is not None:
                app_time_avg = parser.mpi_time_df['app_time'].mean()
                mpi_time_avg = parser.mpi_time_df['mpi_time'].mean()
            else:
                failed_subdirs.append(f"{file_info['subdir_name']}: mpi_time_df is None")
                continue
            
            # Extract total messages sent from aggregate sent statistics
            total_messages_sent = 0
            total_bytes = 0
            if parser.aggregate_sent_df is not None:
                total_messages_sent = parser.aggregate_sent_df['count'].sum()
                total_bytes = parser.aggregate_sent_df['total_bytes'].sum()
            
            # Store metrics with inhibitor info
            self._runs.append({
                "App": file_info['app'],
                "Num Nodes": file_info['num_nodes'],
                "Inhib Message Size": file_info['inhib_message_size'],
                "Inhib Wait Time (us)": file_info['inhib_wait_time_us'],
                "Inhib Comm Sparsity": file_info['inhib_comm_sparsity'],
                "Run Iteration": file_info['run_iteration'],
                "App Time Average": app_time_avg,
                "MPI Time Average": mpi_time_avg,
                "Total Messages Sent": total_messages_sent,
                "Total Bytes Sent": total_bytes
            })
            parsed_subdirs += 1
        
        # Check for failures and raise error if any
        if failed_subdirs:
            error_msg = f"Failed to parse {len(failed_subdirs)} out of {total_subdirs} subdirectories:\n"
            for failure in failed_subdirs:
                error_msg += f"  - {failure}\n"
            raise ValueError(error_msg)
        
        # Convert to DataFrame
        df = pd.DataFrame(self._runs)
        
        return df
    
    def parse_inhibitor_coscheduled_allProfiles(self, directory: Union[str, Path], progress=True) -> pd.DataFrame:
        """
        Parse an inhibitor coscheduled experiment directory and extract metrics for both app and inhibitor including total_bytes.
        
        The directory structure is expected to be:
        <directory>/
            <numNodes>_<app>_inhib_<Inhib Message Size>_<Inhib Wait Time (us)>_d_<Inhib Comm Sparsity>_<runIteration>/
                mpip_profiles/
                    <app_exec>.mpiP
                    <inhib_exec>.mpiP
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
            - "Run Iteration": Run iteration number
            - "App Time Average": Average AppTime across all tasks for App (seconds)
            - "MPI Time Average": Average MPITime across all tasks for App (seconds)
            - "Total Messages Sent": Sum of all messages sent for App
            - "Total Bytes Sent": Sum of all total_bytes from aggregate_sent_df for App
            - "Inhib App Time Average": Average AppTime across all tasks for Inhibitor (seconds)
            - "Inhib MPI Time Average": Average MPITime across all tasks for Inhibitor (seconds)
            - "Inhib Total Messages Sent": Sum of all messages sent for Inhibitor
            - "Inhib Total Bytes Sent": Sum of all total_bytes from aggregate_sent_df for Inhibitor
            
        Raises:
            ValueError: If any subdirectory fails to parse correctly.
        """
        directory = Path(directory)
        self.base_path = directory
        self._runs = []
        
        total_subdirs = 0
        parsed_subdirs = 0
        failed_subdirs = []
        collected_files = []
        
        for subdir in tqdm(directory.iterdir(), desc="Phase 1: Collecting files", disable=not progress):
            if not subdir.is_dir():
                continue
            
            total_subdirs += 1
            
            match = re.match(
                r'^(\d+)_(.+)_inhib_(\d+)_(\d+)_d_(\d+\.?\d*)_(\d+)$',
                subdir.name
            )
            if not match:
                failed_subdirs.append(f"{subdir.name}: failed to match directory pattern")
                continue
            
            num_nodes = int(match.group(1))
            app = match.group(2)
            inhib_message_size = int(match.group(3))
            inhib_wait_time_us = int(match.group(4))
            inhib_comm_sparsity = float(match.group(5))
            run_iteration = int(match.group(6))
            
            mpip_profiles_dir = subdir / "mpip_profiles"
            if not mpip_profiles_dir.exists():
                failed_subdirs.append(f"{subdir.name}: mpip_profiles directory not found")
                print(f"{subdir.name}: mpip_profiles directory not found")
                continue
            
            mpiP_files = list(mpip_profiles_dir.glob("*.mpiP"))
            if len(mpiP_files) != 2:
                failed_subdirs.append(f"{subdir.name}: expected 2 .mpiP files, found {len(mpiP_files)}")
                print(f"{subdir.name}: expected 2 .mpiP files, found {len(mpiP_files)}")
                continue
            
            collected_files.append({
                'subdir_name': subdir.name,
                'mpiP_files': mpiP_files,
                'num_nodes': num_nodes,
                'app': app,
                'inhib_message_size': inhib_message_size,
                'inhib_wait_time_us': inhib_wait_time_us,
                'inhib_comm_sparsity': inhib_comm_sparsity,
                'run_iteration': run_iteration
            })
        
        for file_info in tqdm(collected_files, desc="Phase 2: Parsing files", disable=not progress):
            app_metrics = None
            inhib_metrics = None
            app_total_bytes = 0
            inhib_total_bytes = 0
            parse_errors = []
            
            for mpiP_file in file_info['mpiP_files']:
                filename = mpiP_file.stem
                
                if 'inhib' in filename.lower():
                    parser = MPIPParser(mpiP_file)
                    
                    if parser.mpi_time_df is not None:
                        inhib_app_time_avg = parser.mpi_time_df['app_time'].mean()
                        inhib_mpi_time_avg = parser.mpi_time_df['mpi_time'].mean()
                    else:
                        parse_errors.append(f"mpi_time_df is None for '{filename}'")
                        continue
                    
                    inhib_total_messages_sent = 0
                    inhib_total_bytes = 0
                    if parser.aggregate_sent_df is not None:
                        inhib_total_messages_sent = int(parser.aggregate_sent_df['count'].sum())
                        inhib_total_bytes = parser.aggregate_sent_df['total_bytes'].sum()
                    
                    inhib_metrics = {
                        'app_time_average': inhib_app_time_avg,
                        'mpi_time_average': inhib_mpi_time_avg,
                        'total_messages_sent': inhib_total_messages_sent
                    }
                else:
                    parser = MPIPParser(mpiP_file)
                    
                    if parser.mpi_time_df is not None:
                        app_time_avg = parser.mpi_time_df['app_time'].mean()
                        mpi_time_avg = parser.mpi_time_df['mpi_time'].mean()
                    else:
                        parse_errors.append(f"mpi_time_df is None for '{filename}'")
                        continue
                    
                    total_messages_sent = 0
                    total_bytes = 0
                    if parser.aggregate_sent_df is not None:
                        total_messages_sent = int(parser.aggregate_sent_df['count'].sum())
                        total_bytes = parser.aggregate_sent_df['total_bytes'].sum()
                    
                    app_metrics = {
                        'app_time_average': app_time_avg,
                        'mpi_time_average': mpi_time_avg,
                        'total_messages_sent': total_messages_sent
                    }
                    app_total_bytes = total_bytes
            
            if parse_errors:
                failed_subdirs.append(f"{file_info['subdir_name']}: {'; '.join(parse_errors)}")
                continue
            
            if app_metrics is not None and inhib_metrics is not None:
                self._runs.append({
                    "App": file_info['app'],
                    "Num Nodes": file_info['num_nodes'],
                    "Inhib Message Size": file_info['inhib_message_size'],
                    "Inhib Wait Time (us)": file_info['inhib_wait_time_us'],
                    "Inhib Comm Sparsity": file_info['inhib_comm_sparsity'],
                    "Run Iteration": file_info['run_iteration'],
                    "App Time Average": app_metrics['app_time_average'],
                    "MPI Time Average": app_metrics['mpi_time_average'],
                    "Total Messages Sent": app_metrics['total_messages_sent'],
                    "Total Bytes Sent": app_total_bytes,
                    "Inhib App Time Average": inhib_metrics['app_time_average'],
                    "Inhib MPI Time Average": inhib_metrics['mpi_time_average'],
                    "Inhib Total Messages Sent": inhib_metrics['total_messages_sent'],
                    "Inhib Total Bytes Sent": inhib_total_bytes
                })
                parsed_subdirs += 1
            else:
                missing = []
                if app_metrics is None:
                    missing.append("app")
                if inhib_metrics is None:
                    missing.append("inhib")
                failed_subdirs.append(f"{file_info['subdir_name']}: could not find metrics for {', '.join(missing)}")
                print(f"{file_info['subdir_name']}: could not find metrics for {', '.join(missing)}")
        
        if failed_subdirs:
            error_msg = f"Failed to parse {len(failed_subdirs)} out of {total_subdirs} subdirectories:\n"
            for failure in failed_subdirs:
                error_msg += f"  - {failure}\n"
            raise ValueError(error_msg)
        
        df = pd.DataFrame(self._runs)
        
        return df
