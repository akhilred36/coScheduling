"""
MPIP Parser Module

This module provides classes and utility functions to parse mpiP output files
and store the information in tabular format using pandas.

Supported sections:
1. MPI Time (seconds)
2. Callsite Time statistics
3. Callsite Message Send statistics
4. Callsite I/O statistics
"""

import re
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path


@dataclass
class MPIPHeader:
    """Represents the header information from an mpiP output file."""
    command: Optional[str] = None
    version: Optional[str] = None
    build_date: Optional[str] = None
    start_time: Optional[str] = None
    stop_time: Optional[str] = None
    timer_used: Optional[str] = None
    collector_rank: Optional[int] = None
    collector_pid: Optional[int] = None
    output_dir: Optional[str] = None
    report_generation: Optional[str] = None
    mpi_tasks: List[str] = None
    
    def __post_init__(self):
        if self.mpi_tasks is None:
            self.mpi_tasks = []


class MPIPParser:
    """
    Parser for mpiP output files.
    
    This class parses mpiP output files and provides access to the parsed data
    as pandas DataFrames.
    """
    
    def __init__(self, filepath: Optional[Union[str, Path]] = None):
        """
        Initialize the MPIP parser.
        
        Args:
            filepath: Path to the mpiP output file. If provided, the file will
                     be parsed immediately.
        """
        self.header: Optional[MPIPHeader] = None
        self.mpi_time_df: Optional[pd.DataFrame] = None
        self.callsite_time_df: Optional[pd.DataFrame] = None
        self.callsite_send_df: Optional[pd.DataFrame] = None
        self.callsite_io_df: Optional[pd.DataFrame] = None
        self.aggregate_time_df: Optional[pd.DataFrame] = None
        self.aggregate_sent_df: Optional[pd.DataFrame] = None
        self.aggregate_io_df: Optional[pd.DataFrame] = None
        self.callsite_list_df: Optional[pd.DataFrame] = None
        
        if filepath is not None:
            self.parse(filepath)
    
    def parse(self, filepath: Union[str, Path]) -> 'MPIPParser':
        """
        Parse an mpiP output file.
        
        Args:
            filepath: Path to the mpiP output file.
            
        Returns:
            self: The parser instance.
        """
        filepath = Path(filepath)
        
        with open(filepath, 'r') as f:
            content = f.read()
        
        self._parse_header(content)
        self._parse_callsite_list(content)
        self._parse_mpi_time(content)
        self._parse_aggregate_time(content)
        self._parse_aggregate_sent(content)
        self._parse_aggregate_io(content)
        self._parse_callsite_time_stats(content)
        self._parse_callsite_send_stats(content)
        self._parse_callsite_io_stats(content)
        
        return self
    
    def _parse_header(self, content: str) -> None:
        """Parse the header section of the mpiP output."""
        self.header = MPIPHeader()
        
        lines = content.split('\n')
        
        for line in lines:
            # Command
            match = re.match(r'^@ Command\s*:\s*(.+)$', line)
            if match:
                self.header.command = match.group(1).strip()
                continue
            
            # Version
            match = re.match(r'^@ Version\s*:\s*(.+)$', line)
            if match:
                self.header.version = match.group(1).strip()
                continue
            
            # Build date
            match = re.match(r'^@ MPIP Build date\s*:\s*(.+)$', line)
            if match:
                self.header.build_date = match.group(1).strip()
                continue
            
            # Start time
            match = re.match(r'^@ Start time\s*:\s*(.+)$', line)
            if match:
                self.header.start_time = match.group(1).strip()
                continue
            
            # Stop time
            match = re.match(r'^@ Stop time\s*:\s*(.+)$', line)
            if match:
                self.header.stop_time = match.group(1).strip()
                continue
            
            # Timer used
            match = re.match(r'^@ Timer Used\s*:\s*(.+)$', line)
            if match:
                self.header.timer_used = match.group(1).strip()
                continue
            
            # Collector rank
            match = re.match(r'^@ Collector Rank\s*:\s*(\d+)$', line)
            if match:
                self.header.collector_rank = int(match.group(1))
                continue
            
            # Collector PID
            match = re.match(r'^@ Collector PID\s*:\s*(\d+)$', line)
            if match:
                self.header.collector_pid = int(match.group(1))
                continue
            
            # Output directory
            match = re.match(r'^@ Final Output Dir\s*:\s*(.+)$', line)
            if match:
                self.header.output_dir = match.group(1).strip()
                continue
            
            # Report generation
            match = re.match(r'^@ Report generation\s*:\s*(.+)$', line)
            if match:
                self.header.report_generation = match.group(1).strip()
                continue
            
            # MPI task assignment
            match = re.match(r'^@ MPI Task Assignment\s*:\s*(.+)$', line)
            if match:
                self.header.mpi_tasks.append(match.group(1).strip())
    
    def _parse_callsite_list(self, content: str) -> None:
        """Parse the callsite list section."""
        pattern = r'@--- Callsites:\s*(\d+)---\s*\n-+\s*\n(.*?)@---'
        match = re.search(pattern, content, re.DOTALL)
        
        if match:
            header_line = match.group(0).split('\n')[1]
            # Parse the header to get column names
            header_line = header_line.strip()
            
            # Find the data section
            data_section = match.group(2)
            lines = data_section.strip().split('\n')
            
            data = []
            for line in lines:
                line = line.strip()
                if not line or line.startswith('@'):
                    continue
                
                # Parse: ID Lev File/Address Line Parent_Funct MPI_Call
                # Format: "   1   0 0x751984a12b6c           [unknown]                Alltoall"
                parts = line.split()
                if len(parts) >= 6:
                    row = {
                        'id': int(parts[0]),
                        'level': int(parts[1]),
                        'address': parts[2],
                        'line': parts[3] if parts[3] != '[unknown]' else None,
                        'parent_function': parts[4] if parts[4] != '[unknown]' else None,
                        'mpi_call': parts[5]
                    }
                    data.append(row)
            
            if data:
                self.callsite_list_df = pd.DataFrame(data)
    
    def _parse_mpi_time(self, content: str) -> None:
        """Parse the MPI Time section."""
        pattern = r'@--- MPI Time \(seconds\) .*?\n-+\s*\n(.*?)@---'
        match = re.search(pattern, content, re.DOTALL)
        
        if match:
            data_section = match.group(1)
            lines = data_section.strip().split('\n')
            
            data = []
            for line in lines:
                line = line.strip()
                if not line or line.startswith('@') or 'Task' in line:
                    continue
                
                # Format: "   0     0.0153     0.0142    92.32"
                parts = line.split()
                if len(parts) >= 4:
                    task_id = parts[0]
                    if task_id == '*':      # Skip the aggregate total row
                        continue
                    row = {
                        'task_id': int(task_id) if task_id != '*' else -1,
                        'app_time': float(parts[1]),
                        'mpi_time': float(parts[2]),
                        'mpi_percent': float(parts[3])
                    }
                    data.append(row)
            
            if data:
                self.mpi_time_df = pd.DataFrame(data)
    
    def _parse_aggregate_time(self, content: str) -> None:
        """Parse the Aggregate Time section (top twenty)."""
        pattern = r'@--- Aggregate Time \(top twenty, descending, milliseconds\) .*?\n-+\s*\n(.*?)@---'
        match = re.search(pattern, content, re.DOTALL)
        
        if match:
            data_section = match.group(1)
            lines = data_section.strip().split('\n')
            
            data = []
            for line in lines:
                line = line.strip()
                if not line or line.startswith('@') or 'Call' in line:
                    continue
                
                # Format: "File_open             103       13.5   21.50   23.02          1   0.00"
                parts = line.split()
                if len(parts) >= 7:
                    row = {
                        'mpi_call': parts[0],
                        'site_id': int(parts[1]),
                        'time_ms': float(parts[2]),
                        'app_percent': float(parts[3]),
                        'mpi_percent': float(parts[4]),
                        'count': int(parts[5]),
                        'coverage': float(parts[6])
                    }
                    data.append(row)
            
            if data:
                self.aggregate_time_df = pd.DataFrame(data)
    
    def _parse_aggregate_sent(self, content: str) -> None:
        """Parse the Aggregate Sent Message Size section."""
        pattern = r'@--- Aggregate Sent Message Size \(top twenty, descending, bytes\) .*?\n-+\s*\n(.*?)@---'
        match = re.search(pattern, content, re.DOTALL)
        
        if match:
            data_section = match.group(1)
            lines = data_section.strip().split('\n')
            
            data = []
            for line in lines:
                line = line.strip()
                if not line or line.startswith('@') or 'Call' in line:
                    continue
                
                # Format: "Isend                  77          3   1.04e+05   3.48e+04  25.60"
                parts = line.split()
                if len(parts) >= 6:
                    row = {
                        'mpi_call': parts[0],
                        'site_id': int(parts[1]),
                        'count': int(parts[2]),
                        'total_bytes': float(parts[3]),
                        'avg_bytes': float(parts[4]),
                        'sent_percent': float(parts[5])
                    }
                    data.append(row)
            
            if data:
                self.aggregate_sent_df = pd.DataFrame(data)
    
    def _parse_aggregate_io(self, content: str) -> None:
        """Parse the Aggregate I/O Size section."""
        pattern = r'@--- Aggregate I/O Size \(top twenty, descending, bytes\) .*?\n-+\s*\n(.*?)@---'
        match = re.search(pattern, content, re.DOTALL)
        
        if match:
            data_section = match.group(1)
            lines = data_section.strip().split('\n')
            
            data = []
            for line in lines:
                line = line.strip()
                if not line or line.startswith('@') or 'Call' in line:
                    continue
                
                # Format: "File_read_at          100          1   4.05e+04   4.05e+04  24.23"
                parts = line.split()
                if len(parts) >= 6:
                    row = {
                        'mpi_call': parts[0],
                        'site_id': int(parts[1]),
                        'count': int(parts[2]),
                        'total_bytes': float(parts[3]),
                        'avg_bytes': float(parts[4]),
                        'io_percent': float(parts[5])
                    }
                    data.append(row)
            
            if data:
                self.aggregate_io_df = pd.DataFrame(data)
    
    def _parse_callsite_time_stats(self, content: str) -> None:
        """Parse the Callsite Time statistics section."""
        pattern = r'@--- Callsite Time statistics \(all, milliseconds\): \d+ -+\s*\n-+\s*\n(.*?)@---'
        match = re.search(pattern, content, re.DOTALL)
        
        if match:
            data_section = match.group(1)
            lines = data_section.strip().split('\n')
            
            data = []
            current_mpi_call = None
            
            for line in lines:
                line = line.strip()
                if not line or line.startswith('@'):
                    continue
                
                # Skip header line
                if 'Name' in line or 'Site' in line:
                    continue
                
                parts = line.split()
                if len(parts) >= 9:
                    mpi_call = parts[0]
                    site_id = int(parts[1])
                    # Handle rank which can be '*' for aggregate
                    rank_str = parts[2]
                    if rank_str == '*':      # Skip the aggregate total row
                        continue
                    rank = int(rank_str) if rank_str != '*' else -1
                    
                    row = {
                        'mpi_call': mpi_call,
                        'site_id': site_id,
                        'rank': rank,
                        'count': int(parts[3]),
                        'max_ms': float(parts[4]),
                        'mean_ms': float(parts[5]),
                        'min_ms': float(parts[6]),
                        'app_percent': float(parts[7]),
                        'mpi_percent': float(parts[8])
                    }
                    data.append(row)
            
            if data:
                self.callsite_time_df = pd.DataFrame(data)
    
    def _parse_callsite_send_stats(self, content: str) -> None:
        """Parse the Callsite Message Sent statistics section."""
        pattern = r'@--- Callsite Message Sent statistics \(all, sent bytes\) -+\s*\n-+\s*\n(.*?)@---'
        match = re.search(pattern, content, re.DOTALL)
        
        if match:
            data_section = match.group(1)
            lines = data_section.strip().split('\n')
            
            data = []
            
            for line in lines:
                line = line.strip()
                if not line or line.startswith('@'):
                    continue
                
                # Skip header line
                if 'Name' in line or 'Site' in line:
                    continue
                
                parts = line.split()
                if len(parts) >= 8:
                    mpi_call = parts[0]
                    site_id = int(parts[1])
                    # Handle rank which can be '*' for aggregate
                    rank_str = parts[2]
                    if rank_str == '*':      # Skip the aggregate total row
                        continue
                    rank = int(rank_str) if rank_str != '*' else -1
                    
                    row = {
                        'mpi_call': mpi_call,
                        'site_id': site_id,
                        'rank': rank,
                        'count': int(parts[3]),
                        'max_bytes': float(parts[4]),
                        'mean_bytes': float(parts[5]),
                        'min_bytes': float(parts[6]),
                        'sum_bytes': float(parts[7])
                    }
                    data.append(row)
            
            if data:
                self.callsite_send_df = pd.DataFrame(data)
    
    def _parse_callsite_io_stats(self, content: str) -> None:
        """Parse the Callsite I/O statistics section."""
        pattern = r'@--- Callsite I/O statistics \(all, I/O bytes\) -+\s*\n-+\s*\n(.*?)@---'
        match = re.search(pattern, content, re.DOTALL)
        
        if match:
            data_section = match.group(1)
            lines = data_section.strip().split('\n')
            
            data = []
            
            for line in lines:
                line = line.strip()
                if not line or line.startswith('@'):
                    continue
                
                # Skip header line
                if 'Name' in line or 'Site' in line:
                    continue
                
                parts = line.split()
                if len(parts) >= 8:
                    mpi_call = parts[0]
                    site_id = int(parts[1])
                    # Handle rank which can be '*' for aggregate
                    rank_str = parts[2]
                    if rank_str == '*':      # Skip the aggregate total row
                        continue
                    rank = int(rank_str) if rank_str != '*' else -1
                    
                    row = {
                        'mpi_call': mpi_call,
                        'site_id': site_id,
                        'rank': rank,
                        'count': int(parts[3]),
                        'max_bytes': float(parts[4]),
                        'mean_bytes': float(parts[5]),
                        'min_bytes': float(parts[6]),
                        'sum_bytes': float(parts[7])
                    }
                    data.append(row)
            
            if data:
                self.callsite_io_df = pd.DataFrame(data)
    
    def get_summary(self) -> Dict[str, any]:
        """
        Get a summary of the parsed data.
        
        Returns:
            Dictionary containing summary information about the parsed data.
        """
        summary = {
            'header': {
                'command': self.header.command,
                'version': self.header.version,
                'start_time': self.header.start_time,
                'stop_time': self.header.stop_time,
                'num_mpi_tasks': len(self.header.mpi_tasks) if self.header else 0
            },
            'mpi_time': {
                'num_tasks': len(self.mpi_time_df) if self.mpi_time_df is not None else 0
            },
            'callsite_list': {
                'num_callsites': len(self.callsite_list_df) if self.callsite_list_df is not None else 0
            },
            'aggregate_time': {
                'num_entries': len(self.aggregate_time_df) if self.aggregate_time_df is not None else 0
            },
            'aggregate_sent': {
                'num_entries': len(self.aggregate_sent_df) if self.aggregate_sent_df is not None else 0
            },
            'aggregate_io': {
                'num_entries': len(self.aggregate_io_df) if self.aggregate_io_df is not None else 0
            },
            'callsite_time_stats': {
                'num_entries': len(self.callsite_time_df) if self.callsite_time_df is not None else 0
            },
            'callsite_send_stats': {
                'num_entries': len(self.callsite_send_df) if self.callsite_send_df is not None else 0
            },
            'callsite_io_stats': {
                'num_entries': len(self.callsite_io_df) if self.callsite_io_df is not None else 0
            }
        }
        return summary


def parse_mpiP_file(filepath: Union[str, Path]) -> MPIPParser:
    """
    Convenience function to parse an mpiP output file.
    
    Args:
        filepath: Path to the mpiP output file.
        
    Returns:
        MPIPParser instance with parsed data.
    """
    return MPIPParser(filepath)


def get_mpi_time_df(filepath: Union[str, Path]) -> Optional[pd.DataFrame]:
    """
    Get the MPI Time data as a pandas DataFrame.
    
    Args:
        filepath: Path to the mpiP output file.
        
    Returns:
        DataFrame with MPI Time data, or None if parsing failed.
    """
    parser = MPIPParser(filepath)
    return parser.mpi_time_df


def get_callsite_time_df(filepath: Union[str, Path]) -> Optional[pd.DataFrame]:
    """
    Get the Callsite Time statistics as a pandas DataFrame.
    
    Args:
        filepath: Path to the mpiP output file.
        
    Returns:
        DataFrame with Callsite Time statistics, or None if parsing failed.
    """
    parser = MPIPParser(filepath)
    return parser.callsite_time_df


def get_callsite_send_df(filepath: Union[str, Path]) -> Optional[pd.DataFrame]:
    """
    Get the Callsite Message Send statistics as a pandas DataFrame.
    
    Args:
        filepath: Path to the mpiP output file.
        
    Returns:
        DataFrame with Callsite Message Send statistics, or None if parsing failed.
    """
    parser = MPIPParser(filepath)
    return parser.callsite_send_df


def get_callsite_io_df(filepath: Union[str, Path]) -> Optional[pd.DataFrame]:
    """
    Get the Callsite I/O statistics as a pandas DataFrame.
    
    Args:
        filepath: Path to the mpiP output file.
        
    Returns:
        DataFrame with Callsite I/O statistics, or None if parsing failed.
    """
    parser = MPIPParser(filepath)
    return parser.callsite_io_df


def get_aggregate_time_df(filepath: Union[str, Path]) -> Optional[pd.DataFrame]:
    """
    Get the Aggregate Time statistics as a pandas DataFrame.
    
    Args:
        filepath: Path to the mpiP output file.
        
    Returns:
        DataFrame with Aggregate Time statistics, or None if parsing failed.
    """
    parser = MPIPParser(filepath)
    return parser.aggregate_time_df


def get_aggregate_sent_df(filepath: Union[str, Path]) -> Optional[pd.DataFrame]:
    """
    Get the Aggregate Sent Message Size statistics as a pandas DataFrame.
    
    Args:
        filepath: Path to the mpiP output file.
        
    Returns:
        DataFrame with Aggregate Sent Message Size statistics, or None if parsing failed.
    """
    parser = MPIPParser(filepath)
    return parser.aggregate_sent_df


def get_aggregate_io_df(filepath: Union[str, Path]) -> Optional[pd.DataFrame]:
    """
    Get the Aggregate I/O Size statistics as a pandas DataFrame.
    
    Args:
        filepath: Path to the mpiP output file.
        
    Returns:
        DataFrame with Aggregate I/O Size statistics, or None if parsing failed.
    """
    parser = MPIPParser(filepath)
    return parser.aggregate_io_df


def get_callsite_list_df(filepath: Union[str, Path]) -> Optional[pd.DataFrame]:
    """
    Get the Callsite list as a pandas DataFrame.
    
    Args:
        filepath: Path to the mpiP output file.
        
    Returns:
        DataFrame with Callsite list, or None if parsing failed.
    """
    parser = MPIPParser(filepath)
    return parser.callsite_list_df


def get_header(filepath: Union[str, Path]) -> Optional[MPIPHeader]:
    """
    Get the header information from an mpiP output file.
    
    Args:
        filepath: Path to the mpiP output file.
        
    Returns:
        MPIPHeader object with header information, or None if parsing failed.
    """
    parser = MPIPParser(filepath)
    return parser.header
