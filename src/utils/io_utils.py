"""I/O utility functions."""

import os
from typing import List, Dict, Any, Optional

from ..config import TEXT_COL_CANDIDATES, TITLE_COL_CANDIDATES


def ensure_output_dir(path: str):
    """Ensure an output directory exists.
    
    Args:
        path: Directory path to create
    """
    os.makedirs(path, exist_ok=True)


def pick_col(example: Dict[str, Any], candidates: List[str]) -> Optional[str]:
    """Pick the first available column from a list of candidates.
    
    Args:
        example: Dictionary to search in
        candidates: List of column names to try
        
    Returns:
        First matching column name, or None if none found
    """
    for c in candidates:
        if c in example and example[c] is not None:
            return c
    for k, v in example.items():
        if isinstance(v, str):
            return k
    return None


def get_text_column(example: Dict[str, Any]) -> Optional[str]:
    """Get the text column from an example record.
    
    Args:
        example: Dictionary record
        
    Returns:
        Name of the text column
    """
    return pick_col(example, TEXT_COL_CANDIDATES)


def get_title_column(example: Dict[str, Any]) -> Optional[str]:
    """Get the title column from an example record.
    
    Args:
        example: Dictionary record
        
    Returns:
        Name of the title column
    """
    return pick_col(example, TITLE_COL_CANDIDATES)
