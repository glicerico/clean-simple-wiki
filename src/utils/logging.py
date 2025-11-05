"""Logging utilities for the processing pipeline."""

import os
import json
from typing import Iterable, List, Optional

import orjson

from ..config import LOG_DIR


def ensure_logdir(path: Optional[str] = None):
    """Ensure the log directory exists.
    
    Args:
        path: Directory path to create, defaults to LOG_DIR
    """
    target = path or LOG_DIR
    if target:
        os.makedirs(target, exist_ok=True)


def write_jsonl(path: str, recs: Iterable[dict]):
    """Write records to a JSONL file.
    
    Args:
        path: Output file path
        recs: Iterable of dictionary records
    """
    ensure_logdir(os.path.dirname(path) or ".")
    with open(path, "ab") as f:
        for r in recs:
            f.write(orjson.dumps(r))
            f.write(b"\n")


def reset_log(filename: str) -> Optional[str]:
    """Reset (delete) a log file if it exists.
    
    Args:
        filename: Name of the log file to reset
        
    Returns:
        Full path to the log file, or None if LOG_DIR is not set
    """
    log_dir = LOG_DIR
    if not log_dir:
        return None
    path = os.path.join(log_dir, filename)
    if os.path.exists(path):
        os.remove(path)
    return path


def read_jsonl(path: str) -> List[dict]:
    """Read JSONL file and return list of dicts.
    
    Args:
        path: Path to JSONL file
        
    Returns:
        List of dictionary records
    """
    if not os.path.exists(path):
        return []
    records = []
    with open(path, 'rb') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def append_jsonl(path: str, records: List[dict]):
    """Append records to JSONL file (atomic writes).
    
    Args:
        path: Path to JSONL file
        records: List of dictionary records to append
    """
    ensure_logdir(os.path.dirname(path) or ".")
    with open(path, 'ab') as f:
        for rec in records:
            f.write(orjson.dumps(rec))
            f.write(b'\n')
        f.flush()  # Ensure written to disk
        os.fsync(f.fileno())  # Force OS to write to disk
